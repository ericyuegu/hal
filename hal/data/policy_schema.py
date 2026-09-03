"""Exact compact replay fields for the base action model."""

import hashlib
from collections.abc import Mapping
from typing import Final

import numpy as np

from hal.wire import ACTION_CHANNELS
from hal.wire import MASK_INT32

POLICY_SCHEMA_VERSION: Final[int] = 2

FLOAT_STATE_SUFFIXES: Final[tuple[str, ...]] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "hitlag_left",
)
PACKED_STATE_SUFFIXES: Final[tuple[str, ...]] = (
    "action",
    "stock",
    "jumps_used",
    "hurtbox_state",
    "airborne",
    "direction",
)
PLAYER_PREFIXES: Final[tuple[str, ...]] = ("p1", "p1_nana", "p2", "p2_nana")
LEADER_PREFIXES: Final[tuple[str, ...]] = ("p1", "p2")


def policy_replay_identity(path: str) -> str:
    """Return the stable compact-policy replay ID for a manifest path."""
    # The manifest's 32-bit replay UUID has three known collisions.
    return hashlib.blake2b(
        path.encode("utf-8"),
        digest_size=16,
        person=b"hal-policy-id-v1",
    ).hexdigest()


_FIELDS: Final[dict[str, tuple[int, int, int, int]]] = {
    # Slippi action state is an unsigned 16-bit engine value. Values above the
    # model's 512-row embedding still need to survive the storage round trip.
    # The extra code keeps every uint16 value distinct from missing data.
    "action": (0, 17, 65535, 65536),
    "stock": (17, 3, 4, 7),
    "jumps_used": (20, 4, 8, 15),
    "hurtbox_state": (24, 2, 2, 3),
    "airborne": (26, 2, 1, 3),
}
_DIRECTION_SHIFT: Final[int] = 28
_DIRECTION_MASK: Final[int] = 3
_DIRECTION_TO_CODE: Final[dict[float, int]] = {-1.0: 0, 0.0: 1, 1.0: 2}
_CODE_TO_DIRECTION: Final[np.ndarray] = np.array([-1.0, 0.0, 1.0, np.nan], dtype=np.float32)
_USED_STATE_BITS: Final[int] = (1 << 30) - 1

STICK_SCALE: Final[int] = 80
TRIGGER_SCALE: Final[int] = 140
STICK_SENTINEL: Final[int] = -128
TRIGGER_SENTINEL: Final[int] = 255
BUTTON_SUFFIXES: Final[tuple[str, ...]] = tuple(name.removeprefix("button_") for name in ACTION_CHANNELS[6:])


def _same_shape(values: Mapping[str, np.ndarray], names: tuple[str, ...]) -> tuple[int, ...]:
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"missing compact fields: {missing}")
    shapes = {np.asarray(values[name]).shape for name in names}
    if len(shapes) != 1:
        raise ValueError(f"compact fields have different shapes: {sorted(shapes)}")
    return shapes.pop()


def pack_player_state(values: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = _same_shape(values, PACKED_STATE_SUFFIXES)
    packed = np.zeros(shape, dtype=np.uint32)
    for name, (shift, _width, valid_max, sentinel) in _FIELDS.items():
        source = np.asarray(values[name])
        missing = source == MASK_INT32
        safe = np.where(missing, sentinel, source).astype(np.int64)
        invalid = (~missing) & ((safe < 0) | (safe > valid_max))
        if invalid.any():
            bad = np.unique(source[invalid])
            raise ValueError(
                f"{name} has {int(invalid.sum())} invalid value(s) {bad[:16].tolist()}; "
                f"expected [0, {valid_max}] or MASK_INT32"
            )
        packed |= safe.astype(np.uint32) << shift

    direction = np.asarray(values["direction"], dtype=np.float32)
    codes = np.full(shape, _DIRECTION_MASK, dtype=np.uint32)
    finite = ~np.isnan(direction)
    for value, code in _DIRECTION_TO_CODE.items():
        codes[finite & (direction == value)] = code
    if (finite & (codes == _DIRECTION_MASK)).any():
        bad = np.unique(direction[finite & (codes == _DIRECTION_MASK)])
        raise ValueError(f"direction must be -1, 0, 1, or NaN; got {bad.tolist()}")
    packed |= codes << _DIRECTION_SHIFT
    return packed


def unpack_player_state(packed: np.ndarray) -> dict[str, np.ndarray]:
    value = _validated_player_state(packed)

    out = {name: _unpack_player_state_field(value, name) for name in _FIELDS}
    direction_code = ((value >> _DIRECTION_SHIFT) & _DIRECTION_MASK).astype(np.intp)
    out["direction"] = _CODE_TO_DIRECTION[direction_code]
    return out


def _validated_player_state(packed: np.ndarray) -> np.ndarray:
    value = np.asarray(packed)
    if value.dtype != np.uint32:
        raise TypeError(f"packed state must be uint32, got {value.dtype}")
    if (value & ~np.uint32(_USED_STATE_BITS)).any():
        raise ValueError("packed state uses reserved bits")
    return value


def _unpack_player_state_field(packed: np.ndarray, name: str) -> np.ndarray:
    shift, width, valid_max, sentinel = _FIELDS[name]
    code = ((packed >> shift) & ((1 << width) - 1)).astype(np.int32)
    if ((code > valid_max) & (code != sentinel)).any():
        raise ValueError(f"packed {name} contains a reserved code")
    return np.where(code == sentinel, MASK_INT32, code).astype(np.int32)


def unpack_player_stock(packed: np.ndarray) -> np.ndarray:
    """Decode only stock counts from a packed player-state column."""
    return _unpack_player_state_field(_validated_player_state(packed), "stock")


def _pack_grid(values: np.ndarray, scale: int, sentinel: int, dtype: np.dtype) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    missing = np.isnan(source)
    scaled = np.rint(source * np.float32(scale))
    low = -scale if np.issubdtype(dtype, np.signedinteger) else 0
    if ((~missing) & ((scaled < low) | (scaled > scale))).any():
        raise ValueError(f"controller value is outside [{low / scale}, 1.0]")
    code = np.where(missing, sentinel, scaled).astype(dtype)
    decoded = _unpack_grid(code, scale, sentinel)
    if not np.array_equal(decoded[~missing].view(np.uint32), source[~missing].view(np.uint32)):
        raise ValueError(f"controller value is not exactly on the 1/{scale} grid")
    return code


def _unpack_grid(values: np.ndarray, scale: int, sentinel: int) -> np.ndarray:
    code = np.asarray(values)
    missing = code == sentinel
    out = code.astype(np.float32) / np.float32(scale)
    return np.where(missing, np.float32(np.nan), out).astype(np.float32)


def pack_stick(values: np.ndarray) -> np.ndarray:
    return _pack_grid(values, STICK_SCALE, STICK_SENTINEL, np.dtype(np.int8))


def unpack_stick(values: np.ndarray) -> np.ndarray:
    code = np.asarray(values)
    if code.dtype != np.int8:
        raise TypeError(f"packed stick must be int8, got {code.dtype}")
    valid = (code == STICK_SENTINEL) | ((code >= -STICK_SCALE) & (code <= STICK_SCALE))
    if not valid.all():
        raise ValueError("packed stick contains an invalid code")
    return _unpack_grid(code, STICK_SCALE, STICK_SENTINEL)


def pack_trigger(values: np.ndarray) -> np.ndarray:
    return _pack_grid(values, TRIGGER_SCALE, TRIGGER_SENTINEL, np.dtype(np.uint8))


def unpack_trigger(values: np.ndarray) -> np.ndarray:
    code = np.asarray(values)
    if code.dtype != np.uint8:
        raise TypeError(f"packed trigger must be uint8, got {code.dtype}")
    valid = (code == TRIGGER_SENTINEL) | (code <= TRIGGER_SCALE)
    if not valid.all():
        raise ValueError("packed trigger contains an invalid code")
    return _unpack_grid(code, TRIGGER_SCALE, TRIGGER_SENTINEL)


def pack_buttons(values: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = _same_shape(values, BUTTON_SUFFIXES)
    packed = np.zeros(shape, dtype=np.uint8)
    for bit, name in enumerate(BUTTON_SUFFIXES):
        source = np.asarray(values[name])
        if not np.isin(source, (0, 1)).all():
            raise ValueError(f"button {name} must contain only 0 or 1")
        packed |= source.astype(np.uint8) << bit
    return packed


def unpack_buttons(packed: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(packed)
    if value.dtype != np.uint8:
        raise TypeError(f"packed buttons must be uint8, got {value.dtype}")
    return {name: ((value >> bit) & 1).astype(np.int32) for bit, name in enumerate(BUTTON_SUFFIXES)}


def policy_mds_columns() -> dict[str, str]:
    columns = {
        "policy_schema_version": "int",
        "source_schema_version": "int",
        "replay_id": "str",
        "num_frames": "int",
        "stage": "int",
        "p1_character": "int",
        "p2_character": "int",
        "p1_nana_present": "int",
        "p2_nana_present": "int",
    }
    for prefix in PLAYER_PREFIXES:
        columns.update({f"{prefix}_{name}": "ndarray:float32" for name in FLOAT_STATE_SUFFIXES})
        columns[f"{prefix}_state"] = "ndarray:uint32"
    for prefix in LEADER_PREFIXES:
        for name in ACTION_CHANNELS[:4]:
            columns[f"{prefix}_{name}"] = "ndarray:int8"
        for name in ACTION_CHANNELS[4:6]:
            columns[f"{prefix}_{name}"] = "ndarray:uint8"
        columns[f"{prefix}_buttons"] = "ndarray:uint8"
    return columns


POLICY_MDS_COLUMNS: Final[dict[str, str]] = policy_mds_columns()


def assert_policy_replay_equal(source: Mapping[str, object], compact: Mapping[str, object], where: str) -> None:
    decoded = decode_policy_replay(compact)
    names = ["stage", "p1_character", "p2_character"]
    for prefix in PLAYER_PREFIXES:
        names.extend(f"{prefix}_{name}" for name in (*FLOAT_STATE_SUFFIXES, *PACKED_STATE_SUFFIXES))
    for prefix in LEADER_PREFIXES:
        names.extend(f"{prefix}_{name}" for name in ACTION_CHANNELS[:6])
        names.extend(f"{prefix}_button_{name}" for name in BUTTON_SUFFIXES)
    for name in names:
        expected = np.asarray(source[name])
        actual = np.asarray(decoded[name])
        equal = (
            np.array_equal(expected, actual, equal_nan=True)
            if expected.dtype.kind == "f"
            else np.array_equal(expected, actual)
        )
        if not equal:
            raise ValueError(f"{where}: compact round trip changed {name}")


def _constant_int(source: Mapping[str, object], name: str, frames: int) -> int:
    values = np.asarray(source[name])
    if values.shape != (frames,):
        raise ValueError(f"{name} has shape {values.shape}; expected {(frames,)}")
    if not np.all(values == values[0]):
        raise ValueError(f"{name} is not constant within the replay")
    return int(values[0])


def _nana_is_absent(source: Mapping[str, object], prefix: str) -> bool:
    floats_absent = all(np.isnan(np.asarray(source[f"{prefix}_{name}"])).all() for name in FLOAT_STATE_SUFFIXES)
    ints_absent = all(
        (np.asarray(source[f"{prefix}_{name}"]) == MASK_INT32).all()
        for name in PACKED_STATE_SUFFIXES
        if name != "direction"
    )
    direction_absent = np.isnan(np.asarray(source[f"{prefix}_direction"])).all()
    return bool(floats_absent and ints_absent and direction_absent)


def _scalar_int(source: Mapping[str, object], name: str) -> int:
    value = np.asarray(source[name])
    if value.shape:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return int(value.item())


def encode_policy_replay(source: Mapping[str, object], replay_id: str) -> dict[str, object]:
    frame = np.asarray(source["frame"])
    if frame.ndim != 1 or not len(frame):
        raise ValueError(f"frame must be a non-empty vector, got {frame.shape}")
    frames = len(frame)
    source_version = _scalar_int(source, "schema_version")
    out: dict[str, object] = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": source_version,
        "replay_id": replay_id,
        "num_frames": frames,
        "stage": _constant_int(source, "stage", frames),
        "p1_character": _constant_int(source, "p1_character", frames),
        "p2_character": _constant_int(source, "p2_character", frames),
    }

    for prefix in PLAYER_PREFIXES:
        absent = prefix.endswith("_nana") and _nana_is_absent(source, prefix)
        if prefix.endswith("_nana"):
            out[f"{prefix}_present"] = int(not absent)
        state = {name: np.asarray(source[f"{prefix}_{name}"]) for name in PACKED_STATE_SUFFIXES}
        try:
            packed = pack_player_state(state)
        except ValueError as error:
            leader = prefix.removesuffix("_nana")
            character = _constant_int(source, f"{leader}_character", frames)
            raise ValueError(f"{replay_id}: {prefix}_state (character={character}): {error}") from error
        for name in FLOAT_STATE_SUFFIXES:
            values = np.asarray(source[f"{prefix}_{name}"], dtype=np.float32)
            if values.shape != (frames,):
                raise ValueError(f"{prefix}_{name} has shape {values.shape}; expected {(frames,)}")
            out[f"{prefix}_{name}"] = values[:1] if absent else values
        out[f"{prefix}_state"] = packed[:1] if absent else packed

    for prefix in LEADER_PREFIXES:
        for name in ACTION_CHANNELS[:4]:
            key = f"{prefix}_{name}"
            try:
                out[key] = pack_stick(np.asarray(source[key]))
            except ValueError as error:
                raise ValueError(f"{key}: {error}") from error
        for name in ACTION_CHANNELS[4:6]:
            key = f"{prefix}_{name}"
            try:
                out[key] = pack_trigger(np.asarray(source[key]))
            except ValueError as error:
                raise ValueError(f"{key}: {error}") from error
        buttons = {name: np.asarray(source[f"{prefix}_button_{name}"]) for name in BUTTON_SUFFIXES}
        out[f"{prefix}_buttons"] = pack_buttons(buttons)
    return out


def _select_ranges(
    values: np.ndarray,
    frames: int,
    present: bool,
    name: str,
    ranges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    expected = frames if present else 1
    if values.shape != (expected,):
        raise ValueError(f"{name} has shape {values.shape}; expected {(expected,)}")
    length = sum(stop - start for start, stop in ranges)
    if not present:
        return np.full(length, values[0], dtype=values.dtype)
    if len(ranges) == 1:
        start, stop = ranges[0]
        return values[start:stop]
    return np.concatenate([values[start:stop] for start, stop in ranges])


def _policy_frames(source: Mapping[str, object]) -> int:
    if _scalar_int(source, "policy_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"policy schema version {source['policy_schema_version']} != expected {POLICY_SCHEMA_VERSION}"
        )
    frames = _scalar_int(source, "num_frames")
    if frames < 1:
        raise ValueError(f"num_frames must be positive, got {frames}")
    return frames


def decode_policy_replay_slices(
    source: Mapping[str, object], ranges: tuple[tuple[int, int], ...]
) -> tuple[dict[str, np.ndarray | int], ...]:
    """Decode only the requested real-frame ranges from one compact replay."""
    frames = _policy_frames(source)
    for start, stop in ranges:
        if not 0 <= start <= stop <= frames:
            raise ValueError(f"slice [{start}, {stop}) is outside replay length {frames}")
    if not ranges:
        return ()
    lengths = [stop - start for start, stop in ranges]
    outs: list[dict[str, np.ndarray | int]] = []
    for (start, stop), length in zip(ranges, lengths, strict=True):
        out: dict[str, np.ndarray | int] = {
            "schema_version": _scalar_int(source, "source_schema_version"),
            "frame": np.arange(start, stop, dtype=np.int32),
        }
        for name in ("stage", "p1_character", "p2_character"):
            out[name] = np.full(length, _scalar_int(source, name), dtype=np.int32)
        outs.append(out)

    def assign(name: str, values: np.ndarray) -> None:
        offset = 0
        for out, length in zip(outs, lengths, strict=True):
            out[name] = values[offset : offset + length]
            offset += length

    for prefix in PLAYER_PREFIXES:
        present = True
        if prefix.endswith("_nana"):
            flag = _scalar_int(source, f"{prefix}_present")
            if flag not in (0, 1):
                raise ValueError(f"{prefix}_present must be 0 or 1, got {flag}")
            present = bool(flag)
        for name in FLOAT_STATE_SUFFIXES:
            key = f"{prefix}_{name}"
            values = _select_ranges(np.asarray(source[key], dtype=np.float32), frames, present, key, ranges)
            assign(key, values)
        state_key = f"{prefix}_state"
        packed = _select_ranges(np.asarray(source[state_key]), frames, present, state_key, ranges)
        for name, values in unpack_player_state(packed).items():
            assign(f"{prefix}_{name}", values)

    for prefix in LEADER_PREFIXES:
        for name in ACTION_CHANNELS[:4]:
            key = f"{prefix}_{name}"
            values = np.asarray(source[key])
            if values.shape != (frames,):
                raise ValueError(f"{key} has shape {values.shape}; expected {(frames,)}")
            assign(key, unpack_stick(_select_ranges(values, frames, True, key, ranges)))
        for name in ACTION_CHANNELS[4:6]:
            key = f"{prefix}_{name}"
            values = np.asarray(source[key])
            if values.shape != (frames,):
                raise ValueError(f"{key} has shape {values.shape}; expected {(frames,)}")
            assign(key, unpack_trigger(_select_ranges(values, frames, True, key, ranges)))
        button_key = f"{prefix}_buttons"
        packed_buttons = np.asarray(source[button_key])
        if packed_buttons.shape != (frames,):
            raise ValueError(f"{button_key} has shape {packed_buttons.shape}; expected {(frames,)}")
        selected_buttons = _select_ranges(packed_buttons, frames, True, button_key, ranges)
        for name, values in unpack_buttons(selected_buttons).items():
            assign(f"{prefix}_button_{name}", values)
    return tuple(outs)


def decode_policy_replay_slice(source: Mapping[str, object], start: int, stop: int) -> dict[str, np.ndarray | int]:
    """Decode the real frames in ``[start, stop)`` from one compact replay."""
    return decode_policy_replay_slices(source, ((start, stop),))[0]


def decode_policy_replay(source: Mapping[str, object]) -> dict[str, np.ndarray | int]:
    """Decode every frame from one compact replay."""
    return decode_policy_replay_slice(source, 0, _policy_frames(source))
