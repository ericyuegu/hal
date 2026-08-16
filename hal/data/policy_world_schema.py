"""Compact policy replay storage with ranked tiers and v7 item state.

``mds-policy-world-v7`` is a projection of canonical schema v7, not a new
canonical replay schema.  It preserves the complete compact policy view plus
the four global item slots and per-player ranks.  Extended player dynamics
(``misc_as``, velocities, flags, etc.) remain intentionally out of scope.
"""

from collections.abc import Mapping
from typing import Final

import numpy as np

from hal.data.policy_schema import FLOAT_STATE_SUFFIXES
from hal.data.policy_schema import PLAYER_PREFIXES
from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import decode_policy_replay_slices
from hal.data.policy_schema import encode_policy_replay
from hal.data.schema import Rank
from hal.wire import ITEM_FIELD_SUFFIXES
from hal.wire import ITEM_OWNER_NONE
from hal.wire import ITEM_SLOTS
from hal.wire import MASK_INT32
from hal.wire import item_column

POLICY_WORLD_SCHEMA_VERSION: Final[int] = 1

_TYPE_BITS: Final[int] = 17
_TYPE_MASK: Final[int] = (1 << _TYPE_BITS) - 1
_TYPE_MISSING: Final[int] = 1 << 16
_STATE_SHIFT: Final[int] = _TYPE_BITS
_STATE_BITS: Final[int] = 9
_STATE_MASK: Final[int] = (1 << _STATE_BITS) - 1
_STATE_MISSING: Final[int] = 1 << 8
_OWNER_SHIFT: Final[int] = _STATE_SHIFT + _STATE_BITS
_OWNER_BITS: Final[int] = 3
_OWNER_MASK: Final[int] = (1 << _OWNER_BITS) - 1
_USED_META_BITS: Final[int] = (1 << (_OWNER_SHIFT + _OWNER_BITS)) - 1

ITEM_FLOAT_SUFFIXES: Final[tuple[str, ...]] = ("pos_x", "pos_y", "vel_x", "vel_y")
ITEM_INT_SUFFIXES: Final[tuple[str, ...]] = ("type", "state", "owner")
POLICY_WORLD_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    *(f"{prefix}_{suffix}" for prefix in PLAYER_PREFIXES for suffix in FLOAT_STATE_SUFFIXES),
    *(item_column(slot, suffix) for slot in range(ITEM_SLOTS) for suffix in ITEM_FLOAT_SUFFIXES),
)


def policy_world_mds_columns() -> dict[str, str]:
    columns = dict(POLICY_MDS_COLUMNS)
    columns["policy_world_schema_version"] = "int"
    columns["p1_rank"] = "int"
    columns["p2_rank"] = "int"
    columns["item_present"] = "ndarray:uint8"
    for slot in range(ITEM_SLOTS):
        columns[f"item{slot}_meta"] = "ndarray:uint32"
        for suffix in ITEM_FLOAT_SUFFIXES:
            columns[item_column(slot, suffix)] = "ndarray:float32"
    return columns


POLICY_WORLD_MDS_COLUMNS: Final[dict[str, str]] = policy_world_mds_columns()


def _scalar_int(source: Mapping[str, object], name: str) -> int:
    value = np.asarray(source[name])
    if value.shape:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return int(value.item())


def _constant_int(source: Mapping[str, object], name: str, frames: int) -> int:
    values = np.asarray(source[name])
    if values.shape != (frames,):
        raise ValueError(f"{name} has shape {values.shape}; expected {(frames,)}")
    if not np.all(values == values[0]):
        raise ValueError(f"{name} is not constant within the replay")
    return int(values[0])


def _is_masked(values: np.ndarray) -> np.ndarray:
    return np.isnan(values) if values.dtype.kind == "f" else values == MASK_INT32


def _owner_to_code(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    codes = np.zeros(values.shape, dtype=np.uint32)
    codes[(~missing) & (values == ITEM_OWNER_NONE)] = 1
    for port in range(1, 5):
        codes[(~missing) & (values == port)] = port + 1
    valid = missing | (values == ITEM_OWNER_NONE) | ((values >= 1) & (values <= 4))
    if not valid.all():
        bad = np.unique(values[~valid])
        raise ValueError(f"item owner contains invalid values {bad[:16].tolist()}")
    return codes


def _pack_item_meta(
    item_type: np.ndarray,
    state: np.ndarray,
    owner: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    shape = present.shape
    for name, values in (("type", item_type), ("state", state), ("owner", owner)):
        if values.shape != shape:
            raise ValueError(f"item {name} has shape {values.shape}; expected {shape}")

    type_missing = item_type == MASK_INT32
    state_missing = state == MASK_INT32
    owner_missing = owner == MASK_INT32
    invalid_type = (~type_missing) & ((item_type < 0) | (item_type > 0xFFFF))
    invalid_state = (~state_missing) & ((state < 0) | (state > 0xFF))
    if invalid_type.any():
        raise ValueError(f"item type contains invalid values {np.unique(item_type[invalid_type])[:16].tolist()}")
    if invalid_state.any():
        raise ValueError(f"item state contains invalid values {np.unique(state[invalid_state])[:16].tolist()}")

    type_code = np.where(type_missing, _TYPE_MISSING, item_type).astype(np.uint32)
    state_code = np.where(state_missing, _STATE_MISSING, state).astype(np.uint32)
    owner_code = _owner_to_code(owner, owner_missing)
    packed = type_code | (state_code << _STATE_SHIFT) | (owner_code << _OWNER_SHIFT)
    return np.where(present, packed, 0).astype(np.uint32)


def _unpack_item_meta(meta: np.ndarray, present: np.ndarray) -> dict[str, np.ndarray]:
    if meta.dtype != np.uint32:
        raise TypeError(f"packed item metadata must be uint32, got {meta.dtype}")
    if (meta & ~np.uint32(_USED_META_BITS)).any():
        raise ValueError("packed item metadata uses reserved bits")
    if (meta[~present] != 0).any():
        raise ValueError("absent item slot has nonzero metadata")

    type_code = meta & _TYPE_MASK
    state_code = (meta >> _STATE_SHIFT) & _STATE_MASK
    owner_code = (meta >> _OWNER_SHIFT) & _OWNER_MASK
    if ((type_code > 0xFFFF) & (type_code != _TYPE_MISSING) & present).any():
        raise ValueError("packed item type contains a reserved code")
    if ((state_code > 0xFF) & (state_code != _STATE_MISSING) & present).any():
        raise ValueError("packed item state contains a reserved code")
    if ((owner_code > 5) & present).any():
        raise ValueError("packed item owner contains a reserved code")

    item_type = np.where((~present) | (type_code == _TYPE_MISSING), MASK_INT32, type_code).astype(np.int32)
    state = np.where((~present) | (state_code == _STATE_MISSING), MASK_INT32, state_code).astype(np.int32)
    owner = np.full(meta.shape, MASK_INT32, dtype=np.int32)
    owner[present & (owner_code == 1)] = ITEM_OWNER_NONE
    for port in range(1, 5):
        owner[present & (owner_code == port + 1)] = port
    return {"type": item_type, "state": state, "owner": owner}


def encode_policy_world_replay(source: Mapping[str, object], replay_id: str) -> dict[str, object]:
    out = encode_policy_replay(source, replay_id)
    frames = int(out["num_frames"])
    out["policy_world_schema_version"] = POLICY_WORLD_SCHEMA_VERSION
    for prefix in ("p1", "p2"):
        rank = _constant_int(source, f"{prefix}_rank", frames)
        if rank not in Rank:
            raise ValueError(f"{prefix}_rank has invalid Rank value {rank}")
        out[f"{prefix}_rank"] = rank

    presence = np.zeros(frames, dtype=np.uint8)
    for slot in range(ITEM_SLOTS):
        values = {suffix: np.asarray(source[item_column(slot, suffix)]) for suffix in ITEM_FIELD_SUFFIXES}
        bad_shapes = {name: value.shape for name, value in values.items() if value.shape != (frames,)}
        if bad_shapes:
            raise ValueError(f"item{slot} fields have invalid shapes {bad_shapes}; expected {(frames,)}")
        absent = np.ones(frames, dtype=bool)
        for value in values.values():
            absent &= _is_masked(value)
        present = ~absent
        presence |= present.astype(np.uint8) << slot
        out[f"item{slot}_meta"] = _pack_item_meta(
            values["type"].astype(np.int64),
            values["state"].astype(np.int64),
            values["owner"].astype(np.int64),
            present,
        )
        for suffix in ITEM_FLOAT_SUFFIXES:
            value = values[suffix].astype(np.float32, copy=False)
            if (~present & ~np.isnan(value)).any():
                raise ValueError(f"item{slot}_{suffix} is populated for an absent item slot")
            out[item_column(slot, suffix)] = value
    out["item_present"] = presence
    return out


def _world_frames(source: Mapping[str, object]) -> int:
    version = _scalar_int(source, "policy_world_schema_version")
    if version != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError(f"policy-world schema version {version} != expected {POLICY_WORLD_SCHEMA_VERSION}")
    frames = _scalar_int(source, "num_frames")
    if frames < 1:
        raise ValueError(f"num_frames must be positive, got {frames}")
    return frames


def _select(values: np.ndarray, frames: int, name: str, ranges: tuple[tuple[int, int], ...]) -> np.ndarray:
    if values.shape != (frames,):
        raise ValueError(f"{name} has shape {values.shape}; expected {(frames,)}")
    if len(ranges) == 1:
        start, stop = ranges[0]
        return values[start:stop]
    return np.concatenate([values[start:stop] for start, stop in ranges])


def decode_policy_world_replay_slices(
    source: Mapping[str, object], ranges: tuple[tuple[int, int], ...]
) -> tuple[dict[str, np.ndarray | int], ...]:
    frames = _world_frames(source)
    outs = list(decode_policy_replay_slices(source, ranges))
    if not outs:
        return ()
    lengths = [stop - start for start, stop in ranges]
    boundaries = np.cumsum(lengths)[:-1]

    def assign(name: str, values: np.ndarray) -> None:
        for out, part in zip(outs, np.split(values, boundaries), strict=True):
            out[name] = part

    for prefix in ("p1", "p2"):
        rank = _scalar_int(source, f"{prefix}_rank")
        if rank not in Rank:
            raise ValueError(f"{prefix}_rank has invalid Rank value {rank}")
        for out, length in zip(outs, lengths, strict=True):
            out[f"{prefix}_rank"] = np.full(length, rank, dtype=np.uint8)

    presence = _select(np.asarray(source["item_present"]), frames, "item_present", ranges)
    if presence.dtype != np.uint8:
        raise TypeError(f"item_present must be uint8, got {presence.dtype}")
    if (presence & ~np.uint8((1 << ITEM_SLOTS) - 1)).any():
        raise ValueError("item_present uses reserved bits")
    for slot in range(ITEM_SLOTS):
        present = ((presence >> slot) & 1).astype(bool)
        meta_name = f"item{slot}_meta"
        meta = _select(np.asarray(source[meta_name]), frames, meta_name, ranges)
        for suffix, values in _unpack_item_meta(meta, present).items():
            assign(item_column(slot, suffix), values)
        for suffix in ITEM_FLOAT_SUFFIXES:
            name = item_column(slot, suffix)
            values = _select(np.asarray(source[name], dtype=np.float32), frames, name, ranges)
            if (~present & ~np.isnan(values)).any():
                raise ValueError(f"{name} is populated for an absent item slot")
            assign(name, values)
    return tuple(outs)


def decode_policy_world_replay_slice(
    source: Mapping[str, object], start: int, stop: int
) -> dict[str, np.ndarray | int]:
    return decode_policy_world_replay_slices(source, ((start, stop),))[0]


def decode_policy_world_replay(source: Mapping[str, object]) -> dict[str, np.ndarray | int]:
    return decode_policy_world_replay_slice(source, 0, _world_frames(source))


def policy_row_from_world(source: Mapping[str, object]) -> dict[str, object]:
    """Drop ranks/items without decoding or touching the policy payload."""
    _world_frames(source)
    return {name: source[name] for name in POLICY_MDS_COLUMNS}


def assert_policy_world_replay_equal(source: Mapping[str, object], compact: Mapping[str, object], where: str) -> None:
    decoded = decode_policy_world_replay(compact)
    names = ["p1_rank", "p2_rank"]
    for slot in range(ITEM_SLOTS):
        names.extend(item_column(slot, suffix) for suffix in ITEM_FIELD_SUFFIXES)
    for name in names:
        expected = np.asarray(source[name])
        actual = np.asarray(decoded[name])
        if not np.array_equal(expected, actual, equal_nan=True):
            raise ValueError(f"{where}: compact world round trip changed {name}")
