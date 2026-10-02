"""HAL's 1v1 flat-observation boundary for the native Melee simulator."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import math
import struct
import weakref
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self
from typing import cast

import numpy as np
from numpy.typing import NDArray

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.sim import native_bridge
from hal.wire import ACTION_DIM
from hal.wire import ITEM_SLOTS

_ABI_VERSION = 2
_SNAPSHOT_MAGIC = b"HALMSL02"
_OBSERVATION_SCHEMA = "hal.flat.numeric.v1"
_ACTION_SCHEMA = "hal.controller.v1"
_STAGE_TO_NATIVE = {8: 2, 18: 3, 6: 8, 26: 28, 24: 31, 25: 32}
_PLAYER_FIELDS = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "direction",
    "hitlag_left",
    "action",
    "stock",
    "jumps_used",
    "hurtbox_state",
    "airborne",
)
_ITEM_FIELDS = ("type", "state", "pos_x", "pos_y", "vel_x", "vel_y")
OBSERVATION_FIELDS = (
    "stage",
    "p1_character",
    "p2_character",
    *(f"{prefix}_{name}" for prefix in ("p1", "p1_nana", "p2", "p2_nana") for name in _PLAYER_FIELDS),
    *(f"item{slot}_{name}" for slot in range(ITEM_SLOTS) for name in _ITEM_FIELDS),
)
if len(OBSERVATION_FIELDS) != 71 or set(OBSERVATION_FIELDS) - MDS_PER_FRAME_DTYPES.keys():
    raise RuntimeError("native observation projection no longer matches the HAL schema")

FRAME_DTYPE = np.dtype(
    [
        ("columns", np.uint32, (71,)),
        ("frame_id", np.int32),
        ("applied_action", np.float32, (2, ACTION_DIM)),
        ("wire_inputs", np.int16, (2, 7)),
        ("reward", np.float32, (2,)),
        ("terminated", np.bool_),
        ("truncated", np.bool_),
        ("reset", np.bool_),
        ("_padding", np.uint8),
    ]
)


@dataclass(frozen=True, slots=True)
class NativePlayer:
    port: int
    character: int
    facing: int | None = None
    costume: int = 0
    handicap: int = 9
    start_percent: int = 0


@dataclass(frozen=True, slots=True)
class NativeMatch:
    """HAL stage and 1-based ports; seed is explicit for repeatable rollouts."""

    stage: int
    players: tuple[NativePlayer, NativePlayer]
    seed: int
    max_frame: int = -1
    stocks: int = 4
    damage_ratio: float = 1.0
    ucf_cardinals_1_0_enabled: bool = True


@dataclass(frozen=True, slots=True)
class NativeFrameBatch:
    """Borrowed arrays, overwritten by the next reset, step, or restore."""

    columns: dict[str, NDArray]
    frame_id: NDArray[np.int32]
    applied_action: NDArray[np.float32]
    wire_inputs: NDArray[np.int16]
    reward: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    reset: NDArray[np.bool_]
    records: NDArray | None = None


def _manifest_sha256(data_dir: Path) -> str:
    nested = data_dir / "raw" / "manifest.json"
    manifest = nested if nested.is_file() else data_dir / "manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _config_dict(config: NativeMatch) -> dict[str, object]:
    return asdict(config)


def _read_config(value: object) -> NativeMatch:
    if not isinstance(value, dict):
        raise ValueError("native snapshot has an invalid match configuration")
    fields = cast(dict[str, object], value)
    if set(fields) != {
        "stage",
        "players",
        "seed",
        "max_frame",
        "stocks",
        "damage_ratio",
        "ucf_cardinals_1_0_enabled",
    }:
        raise ValueError("native snapshot has an invalid match configuration")
    players = fields["players"]
    if not isinstance(players, list) or len(players) != 2 or not all(isinstance(player, dict) for player in players):
        raise ValueError("native snapshot has invalid players")

    def integer(field: str) -> int:
        item = fields[field]
        if type(item) is not int:
            raise ValueError(f"native snapshot {field} must be an integer")
        return item

    def player(index: int) -> NativePlayer:
        source = cast(dict[str, object], players[index])
        if set(source) != {"port", "character", "facing", "costume", "handicap", "start_percent"}:
            raise ValueError("native snapshot has invalid player fields")
        if any(
            type(source[name]) is not int for name in ("port", "character", "costume", "handicap", "start_percent")
        ):
            raise ValueError("native snapshot has invalid player values")
        facing = source["facing"]
        if facing is not None and type(facing) is not int:
            raise ValueError("native snapshot has invalid facing")
        return NativePlayer(
            port=cast(int, source["port"]),
            character=cast(int, source["character"]),
            facing=facing,
            costume=cast(int, source["costume"]),
            handicap=cast(int, source["handicap"]),
            start_percent=cast(int, source["start_percent"]),
        )

    damage_ratio = fields["damage_ratio"]
    ucf = fields["ucf_cardinals_1_0_enabled"]
    if type(damage_ratio) not in (int, float) or type(ucf) is not bool:
        raise ValueError("native snapshot has invalid match settings")
    result = NativeMatch(
        stage=integer("stage"),
        players=(player(0), player(1)),
        seed=integer("seed"),
        max_frame=integer("max_frame"),
        stocks=integer("stocks"),
        damage_ratio=cast(int | float, damage_ratio),
        ucf_cardinals_1_0_enabled=ucf,
    )
    _validate_match(result)
    return result


def _validate_match(config: NativeMatch) -> None:
    if type(config.stage) is not int or config.stage not in _STAGE_TO_NATIVE:
        raise ValueError(f"unsupported HAL stage {config.stage!r}")
    if type(config.seed) is not int or not 0 <= config.seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a uint32")
    if type(config.max_frame) is not int or not (config.max_frame == -1 or 0 <= config.max_frame <= 0x7FFFFFFF):
        raise ValueError("max_frame must be -1 or a nonnegative int32")
    if type(config.stocks) is not int or not 1 <= config.stocks <= 255:
        raise ValueError("stocks must be in 1..255")
    if (
        not isinstance(config.damage_ratio, (int, float))
        or not math.isfinite(config.damage_ratio)
        or config.damage_ratio <= 0
    ):
        raise ValueError("damage_ratio must be finite and positive")
    if type(config.ucf_cardinals_1_0_enabled) is not bool:
        raise ValueError("ucf_cardinals_1_0_enabled must be bool")
    if len(config.players) != 2 or {player.port for player in config.players} != {1, 2}:
        raise ValueError("players must use HAL ports 1 and 2 exactly once")
    for player in config.players:
        if type(player.port) is not int or type(player.character) is not int:
            raise ValueError("port and character must be integers")
        if player.facing is not None and player.facing not in (-1, 1):
            raise ValueError("facing must be -1, 1, or None")
        if type(player.costume) is not int or not 0 <= player.costume <= 255:
            raise ValueError("costume must be a uint8")
        if type(player.handicap) is not int or not 1 <= player.handicap <= 9:
            raise ValueError("handicap must be in 1..9")
        if type(player.start_percent) is not int or not 0 <= player.start_percent <= 100:
            raise ValueError("start_percent must be in 0..100")


def _mask(mask: NDArray[np.bool_], batch_size: int) -> NDArray[np.bool_]:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (batch_size,):
        raise ValueError(f"mask must be bool[{batch_size}]")
    return mask


def _quantize(actions: NDArray[np.float32]) -> NDArray[np.int16]:
    if actions.dtype != np.float32 or actions.ndim != 3 or actions.shape[1:] != (2, ACTION_DIM):
        raise ValueError("actions must be float32[B,2,14]")
    source = np.ascontiguousarray(actions)
    result = np.empty((*actions.shape[:2], 7), dtype=np.int16)
    lib = native_bridge.library()
    native_bridge.check(
        lib,
        lib.hal_native_quantize(native_bridge.pointer(source), native_bridge.pointer(result), actions.shape[0] * 2),
    )
    return result


class NativeRolloutBatch:
    """Two HAL ports per lane; the caller owns policy streams and frame history.

    ``chunk_frames`` remains accepted for callers of the former ring-buffer
    adapter; current-frame storage is now constant-sized.
    """

    def __init__(self, data_dir: Path, batch_size: int, *, chunk_frames: int = 256) -> None:
        if (
            type(batch_size) is not int
            or not 0 < batch_size <= 0xFFFFFFFF
            or type(chunk_frames) is not int
            or chunk_frames <= 0
        ):
            raise ValueError("batch_size and chunk_frames must be positive integers")
        sim = importlib.import_module("melee_sim")
        if sim.abi_version() != _ABI_VERSION:
            raise RuntimeError(f"melee_sim ABI {_ABI_VERSION} is required")
        self._sim = sim
        self.batch_size = batch_size
        self._manifest_sha = _manifest_sha256(Path(data_dir).expanduser().resolve())
        self._library_sha = hashlib.sha256(Path(sim.native_library_path()).read_bytes()).hexdigest()
        self._configs: list[NativeMatch | None] = [None] * batch_size
        self._active = np.zeros(batch_size, dtype=np.bool_)
        records = np.zeros(batch_size, dtype=FRAME_DTYPE)
        self.frame = NativeFrameBatch(
            columns={
                name: records["columns"][:, i].view(MDS_PER_FRAME_DTYPES[name])
                for i, name in enumerate(OBSERVATION_FIELDS)
            },
            frame_id=records["frame_id"],
            applied_action=records["applied_action"],
            wire_inputs=records["wire_inputs"],
            reward=records["reward"],
            terminated=records["terminated"],
            truncated=records["truncated"],
            reset=records["reset"],
            records=records,
        )
        self._lib = native_bridge.library()
        self._adapter_sha = native_bridge.library_sha256()
        self._handle = ctypes.c_void_p()
        self._step = self._lib.hal_native_step
        self._lib_create(Path(data_dir), sim.native_library_path(), records)
        self._finalizer = weakref.finalize(self, self._lib.hal_native_destroy, self._handle)

    def _lib_create(self, data_dir: Path, library: Path, records: NDArray) -> None:
        root = data_dir.expanduser().resolve()
        raw = root / "raw" if (root / "raw").is_dir() else root
        native_bridge.check(
            self._lib,
            self._lib.hal_native_create(
                str(library).encode(),
                str(raw).encode(),
                self.batch_size,
                native_bridge.pointer(records),
                ctypes.byref(self._handle),
            ),
        )

    def _check_open(self) -> None:
        if not self._handle.value:
            raise RuntimeError("NativeRolloutBatch is closed")

    @property
    def library_sha256(self) -> str:
        return self._library_sha

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha

    def close(self) -> None:
        if self._handle.value:
            self._finalizer()
            self._handle.value = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def _native_config(self, config: NativeMatch) -> native_bridge.MatchConfig:
        _validate_match(config)
        result = native_bridge.MatchConfig()
        result.stage = _STAGE_TO_NATIVE[config.stage]
        result.random_seed = config.seed
        result.max_frame = config.max_frame
        result.damage_ratio = config.damage_ratio
        result.num_players = 2
        result.stocks = config.stocks
        result.ucf_cardinals = config.ucf_cardinals_1_0_enabled
        for roster, player in enumerate(config.players):
            self._sim.Character(player.character)
            result.players[roster] = native_bridge.PlayerConfig(
                player.character,
                -1,
                0 if player.facing is None else player.facing,
                player.port - 1,
                player.costume,
                player.handicap,
                player.start_percent,
            )
        return result

    def reset(self, configs: Sequence[NativeMatch], mask: NDArray[np.bool_]) -> NativeFrameBatch:
        self._check_open()
        selected = np.ascontiguousarray(_mask(mask, self.batch_size))
        if len(configs) != self.batch_size:
            raise ValueError("configs must contain one match per lane")
        native_configs = (native_bridge.MatchConfig * self.batch_size)()
        ids = np.flatnonzero(selected)
        for lane in ids:
            native_configs[int(lane)] = self._native_config(configs[int(lane)])
        native_bridge.check(
            self._lib, self._lib.hal_native_reset(self._handle, native_configs, native_bridge.pointer(selected))
        )
        for lane in ids:
            self._configs[int(lane)] = configs[int(lane)]
            self._active[int(lane)] = True
        return self.frame

    def step(self, actions: NDArray[np.float32], mask: NDArray[np.bool_]) -> NativeFrameBatch:
        self._check_open()
        selected = _mask(mask, self.batch_size)
        if (
            not isinstance(actions, np.ndarray)
            or actions.dtype != np.float32
            or actions.shape != (self.batch_size, 2, ACTION_DIM)
        ):
            raise ValueError(f"actions must be float32[{self.batch_size},2,{ACTION_DIM}]")
        result = self._step(
            self._handle,
            native_bridge.pointer(actions),
            *actions.strides,
            native_bridge.pointer(selected),
            selected.strides[0],
        )
        if result:
            native_bridge.check(self._lib, result)
        return self.frame

    def save(self, lane: int) -> bytes:
        self._check_lane(lane)
        config = self._configs[lane]
        assert config is not None
        header = {
            "abi": _ABI_VERSION,
            "adapter_abi": native_bridge.ABI_VERSION,
            "adapter_sha256": self._adapter_sha,
            "observation_schema": _OBSERVATION_SCHEMA,
            "action_schema": _ACTION_SCHEMA,
            "manifest_sha256": self._manifest_sha,
            "library_sha256": self._library_sha,
            "config": _config_dict(config),
            "frame_id": int(self.frame.frame_id[lane]),
            "applied_action": self.frame.applied_action[lane].tolist(),
            "wire_inputs": self.frame.wire_inputs[lane].tolist(),
            "terminated": bool(self.frame.terminated[lane]),
            "truncated": bool(self.frame.truncated[lane]),
        }
        metadata = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        size = ctypes.c_size_t()
        native_bridge.check(self._lib, self._lib.hal_native_save_size(self._handle, lane, ctypes.byref(size)))
        buffer = ctypes.create_string_buffer(size.value)
        written = ctypes.c_size_t()
        native_bridge.check(
            self._lib, self._lib.hal_native_save(self._handle, lane, buffer, size.value, ctypes.byref(written))
        )
        return _SNAPSHOT_MAGIC + struct.pack("<I", len(metadata)) + metadata + buffer.raw[: written.value]

    def restore(self, lane: int, state: bytes) -> NativeFrameBatch:
        self._check_open()
        if type(lane) is not int or not 0 <= lane < self.batch_size:
            raise ValueError("lane is out of range")
        if (
            not isinstance(state, bytes)
            or not state.startswith(_SNAPSHOT_MAGIC)
            or len(state) < len(_SNAPSHOT_MAGIC) + 4
        ):
            raise ValueError("invalid HAL native snapshot")
        start = len(_SNAPSHOT_MAGIC)
        header_size = struct.unpack_from("<I", state, start)[0]
        end = start + 4 + header_size
        if end >= len(state):
            raise ValueError("truncated HAL native snapshot")
        header = json.loads(state[start + 4 : end])
        if (
            header["abi"] != _ABI_VERSION
            or header.get("adapter_abi") != native_bridge.ABI_VERSION
            or header.get("adapter_sha256") != self._adapter_sha
            or header["observation_schema"] != _OBSERVATION_SCHEMA
            or header["action_schema"] != _ACTION_SCHEMA
            or header["manifest_sha256"] != self._manifest_sha
            or header["library_sha256"] != self._library_sha
        ):
            raise ValueError("HAL native snapshot identity does not match this simulator")
        config = _read_config(header["config"])
        action = np.asarray(header["applied_action"], dtype=np.float32)
        wire = np.asarray(header["wire_inputs"], dtype=np.int16)
        if action.shape != (2, ACTION_DIM) or wire.shape != (2, 7):
            raise ValueError("HAL native snapshot action shape is invalid")
        restored = np.zeros(1, dtype=FRAME_DTYPE)
        restored["applied_action"][0] = action
        restored["wire_inputs"][0] = wire
        restored["terminated"][0] = bool(header["terminated"])
        restored["truncated"][0] = bool(header["truncated"])
        native_config = self._native_config(config)
        payload = ctypes.create_string_buffer(state[end:])
        native_bridge.check(
            self._lib,
            self._lib.hal_native_restore(
                self._handle,
                lane,
                payload,
                len(state) - end,
                ctypes.byref(native_config),
                native_bridge.pointer(restored),
            ),
        )
        self._configs[lane] = config
        self._active[lane] = True
        if self.frame.frame_id[lane] != header["frame_id"]:
            raise ValueError("HAL native snapshot frame mismatch")
        return self.frame

    def _check_lane(self, lane: int) -> None:
        self._check_open()
        if type(lane) is not int or not 0 <= lane < self.batch_size or not self._active[lane]:
            raise ValueError("lane is out of range or has not been reset")
