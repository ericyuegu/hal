"""Seekable block codec for the policy-world-v8 MDS projection."""

import struct
import zlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from typing import Literal

import numpy as np
import zstd  # ty: ignore[unresolved-import]

from hal.data.policy_schema import FLOAT_STATE_SUFFIXES
from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import decode_policy_replay
from hal.data.policy_schema import unpack_buttons
from hal.data.policy_schema import unpack_player_state
from hal.data.policy_schema import unpack_player_stock
from hal.data.policy_schema import unpack_stick
from hal.data.policy_schema import unpack_trigger
from hal.data.policy_world_schema import ITEM_FLOAT_SUFFIXES
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.data.policy_world_schema import unpack_item_meta
from hal.data.policy_world_v8_selection import assign_v8_ranks
from hal.data.policy_world_v8_selection import replay_id_bytes
from hal.data.reward_events import damage_taken
from hal.data.reward_events import match_point_events
from hal.data.reward_events import stock_loss_events
from hal.data.schema import POLICY_WORLD_V8_MDS_COLUMNS
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.wire import ACTION_CHANNELS
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

POLICY_WORLD_V8_SCHEMA_VERSION: Final[int] = 2
POLICY_WORLD_V8_CODEC_VERSION: Final[int] = 1
POLICY_WORLD_V8_BLOCK_FRAMES: Final[int] = 256

type FeatureGroup = Literal["core", "nana", "items"]

_MAGIC: Final[bytes] = b"HALPWV8\0"
_HEADER = struct.Struct("<8sBBHIIIII")
_INDEX = struct.Struct("<BBIIIQ")
_GROUP_CORE: Final[int] = 1
_GROUP_NANA: Final[int] = 2
_GROUP_ITEMS: Final[int] = 3
_GROUP_REWARD: Final[int] = 4
_P1_NANA_PRESENT: Final[int] = 1 << 0
_P2_NANA_PRESENT: Final[int] = 1 << 1
_ITEMS_PRESENT: Final[int] = 1 << 2
_KNOWN_FLAGS: Final[int] = _P1_NANA_PRESENT | _P2_NANA_PRESENT | _ITEMS_PRESENT

REWARD_EVENT_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("frame", "<u4"),
        ("p1_damage_taken", "<f4"),
        ("p2_damage_taken", "<f4"),
        ("flags", "u1"),
    ],
    align=False,
)
P1_STOCK_LOSS: Final[int] = 1 << 0
P2_STOCK_LOSS: Final[int] = 1 << 1
P1_MATCH_POINT: Final[int] = 1 << 2
P2_MATCH_POINT: Final[int] = 1 << 3
_KNOWN_REWARD_FLAGS: Final[int] = P1_STOCK_LOSS | P2_STOCK_LOSS | P1_MATCH_POINT | P2_MATCH_POINT

type FieldSpec = tuple[str, np.dtype]


def _field(name: str, dtype: str) -> FieldSpec:
    return name, np.dtype(dtype)


_CORE_FIELDS: Final[tuple[FieldSpec, ...]] = (
    *(_field(f"{prefix}_{suffix}", "<f4") for prefix in ("p1", "p2") for suffix in FLOAT_STATE_SUFFIXES),
    _field("p1_state", "<u4"),
    _field("p2_state", "<u4"),
    *(_field(f"{prefix}_{name}", "i1") for prefix in ("p1", "p2") for name in ACTION_CHANNELS[:4]),
    *(_field(f"{prefix}_{name}", "u1") for prefix in ("p1", "p2") for name in ACTION_CHANNELS[4:6]),
    _field("p1_buttons", "u1"),
    _field("p2_buttons", "u1"),
)
_NANA_FIELDS: Final[tuple[FieldSpec, ...]] = (
    *(_field(f"{prefix}_{suffix}", "<f4") for prefix in ("p1_nana", "p2_nana") for suffix in FLOAT_STATE_SUFFIXES),
    _field("p1_nana_state", "<u4"),
    _field("p2_nana_state", "<u4"),
)
_ITEM_FIELDS: Final[tuple[FieldSpec, ...]] = (
    _field("item_present", "u1"),
    *(
        spec
        for slot in range(ITEM_SLOTS)
        for spec in (
            _field(f"item{slot}_meta", "<u4"),
            *(_field(item_column(slot, suffix), "<f4") for suffix in ITEM_FLOAT_SUFFIXES),
        )
    ),
)


@dataclass(frozen=True, slots=True)
class _Chunk:
    group: int
    block: int
    compressed_length: int
    crc32: int
    offset: int


@dataclass(frozen=True, slots=True)
class _PayloadIndex:
    payload: bytes
    flags: int
    frames: int
    block_count: int
    event_count: int
    chunks: dict[tuple[int, int], _Chunk]


def _scalar_int(source: Mapping[str, object], name: str) -> int:
    value = np.asarray(source[name])
    if value.shape:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return int(value.item())


def _validated_arrays(
    source: Mapping[str, object],
    fields: tuple[FieldSpec, ...],
    expected_lengths: Mapping[str, int],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, dtype in fields:
        values = np.asarray(source[name])
        expected = expected_lengths[name]
        if values.shape != (expected,):
            raise ValueError(f"{name} has shape {values.shape}; expected {(expected,)}")
        if values.dtype != dtype:
            raise TypeError(f"{name} has dtype {values.dtype}; expected {dtype}")
        arrays[name] = values
    return arrays


def _encode_fields(
    arrays: Mapping[str, np.ndarray],
    fields: tuple[FieldSpec, ...],
    start: int,
    stop: int,
    *,
    nana_present: tuple[bool, bool] | None = None,
) -> bytes:
    blocks: list[bytes] = []
    for name, _dtype in fields:
        if nana_present is None:
            values = arrays[name][start:stop]
        else:
            side = 0 if name.startswith("p1_") else 1
            values = arrays[name][start:stop] if nana_present[side] else arrays[name][:1]
        blocks.append(values.tobytes(order="C"))
    return b"".join(blocks)


def _compress(raw: bytes) -> tuple[bytes, int]:
    return zstd.compress(raw, 3, 1), zlib.crc32(raw) & 0xFFFFFFFF


def _reward_events(source: Mapping[str, object], frames: int) -> np.ndarray:
    damage = []
    stock = []
    match = []
    for prefix in ("p1", "p2"):
        percent = np.asarray(source[f"{prefix}_percent"], dtype=np.float32)
        packed = np.asarray(source[f"{prefix}_state"])
        if percent.shape != (frames,) or packed.shape != (frames,):
            raise ValueError(f"{prefix} reward inputs do not match num_frames={frames}")
        decoded_stock = unpack_player_stock(packed)
        damage.append(damage_taken(percent))
        stock.append(stock_loss_events(decoded_stock))
        match.append(match_point_events(decoded_stock))

    flags = (
        (stock[0].astype(np.uint8) * P1_STOCK_LOSS)
        | (stock[1].astype(np.uint8) * P2_STOCK_LOSS)
        | (match[0].astype(np.uint8) * P1_MATCH_POINT)
        | (match[1].astype(np.uint8) * P2_MATCH_POINT)
    )
    selected = (damage[0] != 0) | (damage[1] != 0) | (flags != 0)
    events = np.empty(int(selected.sum()), dtype=REWARD_EVENT_DTYPE)
    events["frame"] = np.flatnonzero(selected).astype(np.uint32)
    events["p1_damage_taken"] = damage[0][selected]
    events["p2_damage_taken"] = damage[1][selected]
    events["flags"] = flags[selected]
    return events


def _block_span(frames: int, block: int) -> tuple[int, int]:
    start = block * POLICY_WORLD_V8_BLOCK_FRAMES
    return start, min(start + POLICY_WORLD_V8_BLOCK_FRAMES, frames)


def _append_chunk(chunks: list[tuple[int, int, bytes, int]], group: int, block: int, raw: bytes) -> None:
    compressed, crc32 = _compress(raw)
    chunks.append((group, block, compressed, crc32))


def _encode_payload(source: Mapping[str, object], frames: int, nana_present: tuple[bool, bool]) -> bytes:
    core_lengths = {name: frames for name, _dtype in _CORE_FIELDS}
    nana_lengths = {
        name: frames if nana_present[0 if name.startswith("p1_") else 1] else 1 for name, _dtype in _NANA_FIELDS
    }
    item_lengths = {name: frames for name, _dtype in _ITEM_FIELDS}
    core = _validated_arrays(source, _CORE_FIELDS, core_lengths)
    nana = _validated_arrays(source, _NANA_FIELDS, nana_lengths)
    items = _validated_arrays(source, _ITEM_FIELDS, item_lengths)

    item_presence = items["item_present"]
    if (item_presence & ~np.uint8((1 << ITEM_SLOTS) - 1)).any():
        raise ValueError("item_present uses reserved bits")
    flags = (int(nana_present[0]) * _P1_NANA_PRESENT) | (int(nana_present[1]) * _P2_NANA_PRESENT)
    if item_presence.any():
        flags |= _ITEMS_PRESENT

    block_count = (frames + POLICY_WORLD_V8_BLOCK_FRAMES - 1) // POLICY_WORLD_V8_BLOCK_FRAMES
    chunks: list[tuple[int, int, bytes, int]] = []
    for block in range(block_count):
        start, stop = _block_span(frames, block)
        _append_chunk(chunks, _GROUP_CORE, block, _encode_fields(core, _CORE_FIELDS, start, stop))
    for block in range(block_count):
        start, stop = _block_span(frames, block)
        raw = _encode_fields(nana, _NANA_FIELDS, start, stop, nana_present=nana_present)
        _append_chunk(chunks, _GROUP_NANA, block, raw)
    for block in range(block_count):
        start, stop = _block_span(frames, block)
        _append_chunk(chunks, _GROUP_ITEMS, block, _encode_fields(items, _ITEM_FIELDS, start, stop))

    events = _reward_events(source, frames)
    _append_chunk(chunks, _GROUP_REWARD, 0, events.tobytes(order="C"))
    chunk_count = len(chunks)
    data_offset = _HEADER.size + chunk_count * _INDEX.size
    indexes: list[bytes] = []
    payloads: list[bytes] = []
    offset = data_offset
    for group, block, compressed, crc32 in chunks:
        indexes.append(_INDEX.pack(group, 0, block, len(compressed), crc32, offset))
        payloads.append(compressed)
        offset += len(compressed)
    header = _HEADER.pack(
        _MAGIC,
        POLICY_WORLD_V8_CODEC_VERSION,
        flags,
        0,
        frames,
        POLICY_WORLD_V8_BLOCK_FRAMES,
        block_count,
        len(events),
        chunk_count,
    )
    return b"".join((header, *indexes, *payloads))


def encode_policy_world_v8_replay(
    source: Mapping[str, object],
    *,
    source_name: str,
    p1_port: int,
    p2_port: int,
) -> dict[str, object]:
    """Encode one audited policy-world-v7 row as a v8 MDS row."""
    if _scalar_int(source, "policy_world_schema_version") != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError("input row is not policy-world-v7")
    if _scalar_int(source, "policy_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("input row has an unsupported policy schema")
    if _scalar_int(source, "source_schema_version") != SCHEMA_VERSION:
        raise ValueError("input row is not projected from canonical schema v7")
    frames = _scalar_int(source, "num_frames")
    if frames < 1 or frames > np.iinfo(np.uint32).max:
        raise ValueError(f"num_frames is outside uint32: {frames}")
    if p1_port == p2_port or any(port not in (1, 2, 3, 4) for port in (p1_port, p2_port)):
        raise ValueError(f"physical ports must be distinct values in 1..4, got {(p1_port, p2_port)}")
    identity = replay_id_bytes(source["replay_id"] if isinstance(source["replay_id"], bytes | str) else b"")
    ranks = assign_v8_ranks(
        source_name,
        identity,
        _scalar_int(source, "p1_rank"),
        _scalar_int(source, "p2_rank"),
    )
    nana_present = (
        bool(_scalar_int(source, "p1_nana_present")),
        bool(_scalar_int(source, "p2_nana_present")),
    )
    for side, _present in enumerate(nana_present, start=1):
        if _scalar_int(source, f"p{side}_nana_present") not in (0, 1):
            raise ValueError(f"p{side}_nana_present must be 0 or 1")
    stock = (unpack_player_stock(np.asarray(source["p1_state"])), unpack_player_stock(np.asarray(source["p2_state"])))
    mc_terminated = int(any(len(values) == frames and int(values[-1]) == 0 for values in stock))
    scalars = {name: _scalar_int(source, name) for name in ("stage", "p1_character", "p2_character")}
    for name, value in scalars.items():
        if not 0 <= value <= np.iinfo(np.uint8).max:
            raise ValueError(f"{name} is outside uint8: {value}")
    return {
        "block_payload": _encode_payload(source, frames, nana_present),
        "replay_id": identity,
        "num_frames": frames,
        "policy_world_schema_version": POLICY_WORLD_V8_SCHEMA_VERSION,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        **scalars,
        "p1_nana_present": int(nana_present[0]),
        "p2_nana_present": int(nana_present[1]),
        "p1_rank": int(ranks[0]),
        "p2_rank": int(ranks[1]),
        "p1_port": p1_port,
        "p2_port": p2_port,
        "rank_imputed_mask": ranks[2],
        "mc_terminated": mc_terminated,
    }


def _parse_payload(source: Mapping[str, object]) -> _PayloadIndex:
    if set(source) != set(POLICY_WORLD_V8_MDS_COLUMNS):
        missing = sorted(set(POLICY_WORLD_V8_MDS_COLUMNS) - set(source))
        extra = sorted(set(source) - set(POLICY_WORLD_V8_MDS_COLUMNS))
        raise ValueError(f"policy-world-v8 columns differ: missing={missing}, extra={extra}")
    if _scalar_int(source, "policy_world_schema_version") != POLICY_WORLD_V8_SCHEMA_VERSION:
        raise ValueError("policy-world-v8 schema version differs")
    if _scalar_int(source, "policy_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("policy schema version differs")
    if _scalar_int(source, "source_schema_version") != SCHEMA_VERSION:
        raise ValueError("source schema version differs")
    replay_id_bytes(source["replay_id"] if isinstance(source["replay_id"], bytes | str) else b"")
    nana_flags = tuple(_scalar_int(source, f"p{side}_nana_present") for side in (1, 2))
    if any(value not in (0, 1) for value in nana_flags):
        raise ValueError("Nana presence flags must be 0 or 1")
    try:
        Rank(_scalar_int(source, "p1_rank"))
        Rank(_scalar_int(source, "p2_rank"))
    except ValueError as error:
        raise ValueError("policy-world-v8 row has an invalid rank") from error
    ports = (_scalar_int(source, "p1_port"), _scalar_int(source, "p2_port"))
    if ports[0] == ports[1] or any(port not in (1, 2, 3, 4) for port in ports):
        raise ValueError(f"policy-world-v8 row has invalid physical ports {ports}")
    if _scalar_int(source, "rank_imputed_mask") & ~3:
        raise ValueError("rank_imputed_mask uses reserved bits")
    if _scalar_int(source, "mc_terminated") not in (0, 1):
        raise ValueError("mc_terminated must be 0 or 1")
    payload_value = source["block_payload"]
    if not isinstance(payload_value, bytes):
        raise TypeError(f"block_payload must be bytes, got {type(payload_value).__name__}")
    payload = payload_value
    if len(payload) < _HEADER.size:
        raise ValueError("block_payload is shorter than its header")
    magic, version, flags, reserved, frames, block_frames, block_count, event_count, chunk_count = _HEADER.unpack_from(
        payload
    )
    if magic != _MAGIC or version != POLICY_WORLD_V8_CODEC_VERSION:
        raise ValueError("block_payload magic or codec version differs")
    if flags & ~_KNOWN_FLAGS or reserved:
        raise ValueError("block_payload header uses reserved bits")
    outer_frames = _scalar_int(source, "num_frames")
    expected_blocks = (frames + POLICY_WORLD_V8_BLOCK_FRAMES - 1) // POLICY_WORLD_V8_BLOCK_FRAMES
    if frames != outer_frames or frames < 1 or block_frames != POLICY_WORLD_V8_BLOCK_FRAMES:
        raise ValueError("block_payload frame geometry differs from the MDS row")
    if block_count != expected_blocks:
        raise ValueError(f"block_payload declares {block_count} blocks; expected {expected_blocks}")
    if chunk_count != 3 * block_count + 1:
        raise ValueError(f"block_payload declares {chunk_count} chunks; expected {3 * block_count + 1}")
    index_end = _HEADER.size + chunk_count * _INDEX.size
    if index_end > len(payload):
        raise ValueError("block_payload chunk index is truncated")

    chunks: dict[tuple[int, int], _Chunk] = {}
    expected_order = [
        *((_GROUP_CORE, block) for block in range(block_count)),
        *((_GROUP_NANA, block) for block in range(block_count)),
        *((_GROUP_ITEMS, block) for block in range(block_count)),
        (_GROUP_REWARD, 0),
    ]
    next_offset = index_end
    for position, expected_key in enumerate(expected_order):
        group, entry_reserved, block, compressed_length, crc32, offset = _INDEX.unpack_from(
            payload, _HEADER.size + position * _INDEX.size
        )
        key = (group, block)
        if entry_reserved or key != expected_key:
            raise ValueError(f"block_payload chunk {position} is not in group-major order")
        if offset != next_offset or compressed_length < 1 or offset + compressed_length > len(payload):
            raise ValueError(f"block_payload chunk {position} has invalid bounds")
        chunks[key] = _Chunk(group, block, compressed_length, crc32, offset)
        next_offset += compressed_length
    if next_offset != len(payload):
        raise ValueError("block_payload has trailing bytes")

    expected_flags = nana_flags[0] * _P1_NANA_PRESENT | nana_flags[1] * _P2_NANA_PRESENT
    if flags & (_P1_NANA_PRESENT | _P2_NANA_PRESENT) != expected_flags:
        raise ValueError("block_payload Nana flags differ from the MDS row")
    if event_count > frames:
        raise ValueError("block_payload has more reward events than frames")
    return _PayloadIndex(payload, flags, frames, block_count, event_count, chunks)


def _expected_chunk_bytes(index: _PayloadIndex, group: int, block: int) -> int:
    start, stop = _block_span(index.frames, block)
    span = stop - start
    if group == _GROUP_CORE:
        return span * sum(dtype.itemsize for _name, dtype in _CORE_FIELDS)
    if group == _GROUP_NANA:
        p1 = span if index.flags & _P1_NANA_PRESENT else 1
        p2 = span if index.flags & _P2_NANA_PRESENT else 1
        per_side = sum(dtype.itemsize for name, dtype in _NANA_FIELDS if name.startswith("p1_"))
        return (p1 + p2) * per_side
    if group == _GROUP_ITEMS:
        return span * sum(dtype.itemsize for _name, dtype in _ITEM_FIELDS)
    if group == _GROUP_REWARD:
        return index.event_count * REWARD_EVENT_DTYPE.itemsize
    raise AssertionError(f"unknown group {group}")


def _decompress_chunk(index: _PayloadIndex, group: int, block: int) -> bytes:
    chunk = index.chunks[(group, block)]
    compressed = index.payload[chunk.offset : chunk.offset + chunk.compressed_length]
    try:
        raw = zstd.decompress(compressed)
    except zstd.Error as error:
        raise ValueError(f"policy-world-v8 group {group} block {block} failed to decompress") from error
    expected = _expected_chunk_bytes(index, group, block)
    if len(raw) != expected:
        raise ValueError(
            f"policy-world-v8 group {group} block {block} decoded to {len(raw)} bytes; expected {expected}"
        )
    if zlib.crc32(raw) & 0xFFFFFFFF != chunk.crc32:
        raise ValueError(f"policy-world-v8 group {group} block {block} failed CRC32")
    return raw


def _decode_fields(raw: bytes, fields: tuple[FieldSpec, ...], lengths: Mapping[str, int]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    offset = 0
    for name, dtype in fields:
        length = lengths[name]
        size = length * dtype.itemsize
        values[name] = np.frombuffer(raw, dtype=dtype, count=length, offset=offset).copy()
        offset += size
    if offset != len(raw):
        raise AssertionError("fixed field layout did not consume its chunk")
    return values


def _decode_block(index: _PayloadIndex, group: int, block: int) -> dict[str, np.ndarray]:
    start, stop = _block_span(index.frames, block)
    span = stop - start
    if group == _GROUP_CORE:
        fields = _CORE_FIELDS
        lengths = {name: span for name, _dtype in fields}
    elif group == _GROUP_NANA:
        fields = _NANA_FIELDS
        lengths = {
            name: span if index.flags & (_P1_NANA_PRESENT if name.startswith("p1_") else _P2_NANA_PRESENT) else 1
            for name, _dtype in fields
        }
    elif group == _GROUP_ITEMS:
        fields = _ITEM_FIELDS
        lengths = {name: span for name, _dtype in fields}
    else:
        raise AssertionError(f"group {group} is not a frame group")
    return _decode_fields(_decompress_chunk(index, group, block), fields, lengths)


def policy_world_v7_row_from_v8(source: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct the exact compact policy-world-v7 mapping in memory."""
    index = _parse_payload(source)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for group in (_GROUP_CORE, _GROUP_NANA, _GROUP_ITEMS):
        for block in range(index.block_count):
            for name, values in _decode_block(index, group, block).items():
                grouped[name].append(values)

    nana_present = {
        "p1_nana": bool(index.flags & _P1_NANA_PRESENT),
        "p2_nana": bool(index.flags & _P2_NANA_PRESENT),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, blocks in grouped.items():
        nana_prefix = "p1_nana" if name.startswith("p1_nana_") else "p2_nana" if name.startswith("p2_nana_") else None
        if nana_prefix is not None and not nana_present[nana_prefix]:
            first = blocks[0]
            if any(block.tobytes() != first.tobytes() for block in blocks[1:]):
                raise ValueError(f"absent {nana_prefix} singleton differs between blocks")
            arrays[name] = first
        else:
            arrays[name] = np.concatenate(blocks)
    if bool(np.asarray(arrays["item_present"]).any()) != bool(index.flags & _ITEMS_PRESENT):
        raise ValueError("block_payload item-presence flag differs from the item frames")

    out: dict[str, object] = {
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "replay_id": replay_id_bytes(
            source["replay_id"] if isinstance(source["replay_id"], bytes | str) else b""
        ).hex(),
        "num_frames": index.frames,
        "stage": _scalar_int(source, "stage"),
        "p1_character": _scalar_int(source, "p1_character"),
        "p2_character": _scalar_int(source, "p2_character"),
        "p1_nana_present": _scalar_int(source, "p1_nana_present"),
        "p2_nana_present": _scalar_int(source, "p2_nana_present"),
        "p1_rank": _scalar_int(source, "p1_rank"),
        "p2_rank": _scalar_int(source, "p2_rank"),
        **arrays,
    }
    if set(out) != set(POLICY_WORLD_MDS_COLUMNS):
        raise AssertionError("v8 codec did not reconstruct the policy-world-v7 column set")
    return out


def policy_row_from_v8(source: Mapping[str, object]) -> dict[str, object]:
    """Project v8 to ordinary compact mds-policy-v7 in memory."""
    world = policy_world_v7_row_from_v8(source)
    return {name: world[name] for name in POLICY_MDS_COLUMNS}


def decode_policy_world_v8_replay(source: Mapping[str, object]) -> dict[str, np.ndarray | int]:
    """Fully decode v8 to the canonical-v7 policy-world view."""
    return decode_policy_world_replay(policy_world_v7_row_from_v8(source))


def decode_policy_v8_replay(source: Mapping[str, object]) -> dict[str, np.ndarray | int]:
    """Fully decode only the ordinary policy-v7 projection."""
    return decode_policy_replay(policy_row_from_v8(source))


def _slice_compact_group(
    index: _PayloadIndex,
    group: int,
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    first_block = start // POLICY_WORLD_V8_BLOCK_FRAMES
    last_block = (stop - 1) // POLICY_WORLD_V8_BLOCK_FRAMES
    for block in range(first_block, last_block + 1):
        block_start, block_stop = _block_span(index.frames, block)
        local_start = max(start, block_start) - block_start
        local_stop = min(stop, block_stop) - block_start
        for name, values in _decode_block(index, group, block).items():
            if len(values) == 1 and group == _GROUP_NANA:
                part = np.full(local_stop - local_start, values[0], dtype=values.dtype)
            else:
                part = values[local_start:local_stop]
            pieces[name].append(part)
    return {name: np.concatenate(blocks) for name, blocks in pieces.items()}


def decode_policy_world_v8_slice(
    source: Mapping[str, object],
    start: int,
    stop: int,
    *,
    groups: frozenset[FeatureGroup] = frozenset(("core", "nana", "items")),
) -> dict[str, np.ndarray | int]:
    """Decode only intersecting blocks in the requested feature groups."""
    unknown = groups - {"core", "nana", "items"}
    if unknown:
        raise ValueError(f"unknown policy-world-v8 feature groups: {sorted(unknown)}")
    index = _parse_payload(source)
    if not 0 <= start < stop <= index.frames:
        raise ValueError(f"slice [{start}, {stop}) is outside replay length {index.frames}")
    length = stop - start
    out: dict[str, np.ndarray | int] = {
        "schema_version": SCHEMA_VERSION,
        "frame": np.arange(start, stop, dtype=np.int32),
    }
    for name in ("stage", "p1_character", "p2_character"):
        out[name] = np.full(length, _scalar_int(source, name), dtype=np.int32)
    for name in ("p1_rank", "p2_rank", "p1_port", "p2_port"):
        out[name] = np.full(length, _scalar_int(source, name), dtype=np.uint8)
    imputed_mask = _scalar_int(source, "rank_imputed_mask")
    if imputed_mask & ~3:
        raise ValueError("rank_imputed_mask uses reserved bits")
    for side in (1, 2):
        out[f"p{side}_rank_imputed"] = np.full(length, bool(imputed_mask & (1 << (side - 1))), dtype=np.bool_)

    if "core" in groups:
        compact = _slice_compact_group(index, _GROUP_CORE, start, stop)
        for prefix in ("p1", "p2"):
            for suffix in FLOAT_STATE_SUFFIXES:
                out[f"{prefix}_{suffix}"] = compact[f"{prefix}_{suffix}"]
            out.update(
                {
                    f"{prefix}_{name}": values
                    for name, values in unpack_player_state(compact[f"{prefix}_state"]).items()
                }
            )
            for name in ACTION_CHANNELS[:4]:
                out[f"{prefix}_{name}"] = unpack_stick(compact[f"{prefix}_{name}"])
            for name in ACTION_CHANNELS[4:6]:
                out[f"{prefix}_{name}"] = unpack_trigger(compact[f"{prefix}_{name}"])
            out.update(
                {
                    f"{prefix}_button_{name}": values
                    for name, values in unpack_buttons(compact[f"{prefix}_buttons"]).items()
                }
            )
    if "nana" in groups:
        compact = _slice_compact_group(index, _GROUP_NANA, start, stop)
        for prefix in ("p1_nana", "p2_nana"):
            for suffix in FLOAT_STATE_SUFFIXES:
                out[f"{prefix}_{suffix}"] = compact[f"{prefix}_{suffix}"]
            out.update(
                {
                    f"{prefix}_{name}": values
                    for name, values in unpack_player_state(compact[f"{prefix}_state"]).items()
                }
            )
    if "items" in groups:
        compact = _slice_compact_group(index, _GROUP_ITEMS, start, stop)
        presence = compact["item_present"]
        for slot in range(ITEM_SLOTS):
            present = ((presence >> slot) & 1).astype(bool)
            for suffix, values in unpack_item_meta(compact[f"item{slot}_meta"], present).items():
                out[item_column(slot, suffix)] = values
            for suffix in ITEM_FLOAT_SUFFIXES:
                name = item_column(slot, suffix)
                values = compact[name]
                if (~present & ~np.isnan(values)).any():
                    raise ValueError(f"{name} is populated for an absent item slot")
                out[name] = values
    return out


def decode_policy_world_v8_reward_events(source: Mapping[str, object]) -> np.ndarray:
    """Decode and validate the replay-level sparse reward segment."""
    index = _parse_payload(source)
    raw = _decompress_chunk(index, _GROUP_REWARD, 0)
    events = np.frombuffer(raw, dtype=REWARD_EVENT_DTYPE).copy()
    if len(events) != index.event_count:
        raise AssertionError("reward event count changed after fixed-size decode")
    if len(events) and (np.diff(events["frame"].astype(np.int64)) <= 0).any():
        raise ValueError("reward events are not in strictly increasing frame order")
    if len(events) and int(events["frame"][-1]) >= index.frames:
        raise ValueError("reward event frame is outside the replay")
    if (events["flags"] & ~np.uint8(_KNOWN_REWARD_FLAGS)).any():
        raise ValueError("reward event uses reserved flag bits")
    return events
