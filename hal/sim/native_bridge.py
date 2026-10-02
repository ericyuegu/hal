"""Typed loader for HAL's compiled policy boundary; uses only the public simulator ABI."""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ABI_VERSION = 1


class PlayerConfig(ctypes.Structure):
    _fields_ = [
        ("character", ctypes.c_uint8),
        ("team", ctypes.c_int8),
        ("facing", ctypes.c_int8),
        ("controller_port", ctypes.c_int8),
        ("costume", ctypes.c_uint8),
        ("handicap", ctypes.c_uint8),
        ("start_percent", ctypes.c_uint8),
    ]


class MatchConfig(ctypes.Structure):
    _fields_ = [
        ("stage", ctypes.c_uint32),
        ("random_seed", ctypes.c_uint32),
        ("max_frame", ctypes.c_int32),
        ("damage_ratio", ctypes.c_float),
        ("num_players", ctypes.c_uint8),
        ("is_teams", ctypes.c_uint8),
        ("friendly_fire", ctypes.c_uint8),
        ("stocks", ctypes.c_uint8),
        ("viewpoint_player", ctypes.c_uint8),
        ("ucf_cardinals", ctypes.c_uint8),
        ("players", PlayerConfig * 4),
    ]


def pointer(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def library_path() -> Path:
    value = os.environ.get("HAL_NATIVE_LIBRARY")
    return (
        Path(value).expanduser().resolve()
        if value
        else Path(__file__).resolve().parents[2] / "build/native_adapter/libhal_native.so"
    )


def library() -> ctypes.CDLL:
    return _load(library_path())[0]


def library_sha256() -> str:
    return _load(library_path())[1]


@lru_cache(maxsize=4)
def _load(path: Path) -> tuple[ctypes.CDLL, str]:
    # Import here to keep schema generation independent of loading a native build.
    from hal.sim.native_build import schema_hash

    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run python -m hal.sim.native_build --native-source /path/to/melee-sim-light"
        )
    lib = ctypes.CDLL(str(path))
    ptr, size, u32, stride = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_ssize_t
    signatures = {
        "hal_native_abi": ([], u32),
        "hal_native_schema": ([], ctypes.c_char_p),
        "hal_native_frame_size": ([], size),
        "hal_native_error": ([ctypes.c_int], ctypes.c_char_p),
        "hal_native_create": ([ctypes.c_char_p, ctypes.c_char_p, u32, ptr, ctypes.POINTER(ptr)], ctypes.c_int),
        "hal_native_destroy": ([ptr], None),
        "hal_native_reset": ([ptr, ptr, ptr], ctypes.c_int),
        "hal_native_step": ([ptr, ptr, stride, stride, stride, ptr, stride], ctypes.c_int),
        "hal_native_quantize": ([ptr, ptr, size], ctypes.c_int),
        "hal_native_save_size": ([ptr, u32, ctypes.POINTER(size)], ctypes.c_int),
        "hal_native_save": ([ptr, u32, ptr, size, ctypes.POINTER(size)], ctypes.c_int),
        "hal_native_restore": ([ptr, u32, ptr, size, ptr, ptr], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes = arguments
        function.restype = result
    if lib.hal_native_abi() != ABI_VERSION or lib.hal_native_schema().decode() != schema_hash():
        raise RuntimeError("HAL native adapter ABI/schema mismatch; rebuild the adapter")
    if lib.hal_native_frame_size() != 440 or ctypes.sizeof(MatchConfig) != 52:
        raise RuntimeError("HAL native adapter layout mismatch")
    return lib, hashlib.sha256(path.read_bytes()).hexdigest()


def check(lib: ctypes.CDLL, result: int) -> None:
    if result:
        message = lib.hal_native_error(result).decode()
        if result in (1, 4, 10, 11, 12, 13, 14):
            raise ValueError(message)
        raise RuntimeError(message)
