"""Unit tests for hal.data.extract — pure-function helpers, no .slp fixture."""

from pathlib import Path

import numpy as np
import pytest

from hal.data.extract import _action_frame_from_states
from hal.data.extract import _list_to_np
from hal.data.extract import _unpack_buttons
from hal.data.extract import extract_replay
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.paths import DEV_ARCHIVE_PATH
from hal.wire import BUTTON_BITS
from hal.wire import MASK_INT32
from hal.wire import mask_value as _mask_value


class _ArrowLike:
    """Minimal stand-in for a pyarrow Array exposing only ``.to_pylist()``."""

    def __init__(self, values: list[object]) -> None:
        self._values = values

    def to_pylist(self) -> list[object]:
        return list(self._values)


def test_mask_value_float_is_nan() -> None:
    assert np.isnan(_mask_value(np.float32))
    assert np.isnan(_mask_value(np.float64))


def test_mask_value_signed_int_is_dtype_min_for_narrow() -> None:
    assert _mask_value(np.int8) == np.iinfo(np.int8).min
    assert _mask_value(np.int16) == np.iinfo(np.int16).min


def test_mask_value_int32_is_np_mask_value() -> None:
    assert _mask_value(np.int32) == MASK_INT32
    assert _mask_value(np.int32) == (1 << 31) - 1


def test_mask_value_unsigned_is_dtype_max() -> None:
    assert _mask_value(np.uint8) == np.iinfo(np.uint8).max
    assert _mask_value(np.uint16) == np.iinfo(np.uint16).max


def test_mask_value_float_callers_must_use_isnan_not_eq() -> None:
    """Regression for the silent-mask-detection footgun: ``arr == mask`` is
    always False for the NaN sentinel because ``nan != nan``."""
    mask = _mask_value(np.float32)
    arr = np.full(5, mask, dtype=np.float32)
    assert not np.any(arr == mask)  # the trap
    assert np.all(np.isnan(arr))  # the correct check


def test_action_frame_from_states_run_length() -> None:
    # state 1 for 3 frames, state 2 for 2 frames, state 3 for 1 frame, state 1 for 2.
    states = [1, 1, 1, 2, 2, 3, 1, 1]
    result = _action_frame_from_states(states)
    assert list(result) == [1, 2, 3, 1, 2, 1, 1, 2]
    assert result.dtype == np.int32


def test_action_frame_from_states_empty_on_none() -> None:
    result = _action_frame_from_states(None)
    assert result.shape == (0,)
    assert result.dtype == np.int32


def test_action_frame_from_states_single_frame() -> None:
    assert list(_action_frame_from_states([7])) == [1]


def test_unpack_buttons_decodes_bitmask() -> None:
    a_bit = BUTTON_BITS["a"]
    b_bit = BUTTON_BITS["b"]
    # Frame 0: A only; frame 1: A+B; frame 2: none.
    physical = _ArrowLike([a_bit, a_bit | b_bit, 0])
    out = _unpack_buttons(physical, length=3)
    assert list(out["a"]) == [1, 1, 0]
    assert list(out["b"]) == [0, 1, 0]
    assert list(out["x"]) == [0, 0, 0]
    assert out["a"].dtype == np.int32


def test_unpack_buttons_none_yields_all_zeros() -> None:
    out = _unpack_buttons(None, length=4)
    assert set(out) == set(BUTTON_BITS)
    for arr in out.values():
        assert list(arr) == [0, 0, 0, 0]


def test_list_to_np_substitutes_none_with_mask() -> None:
    arr = _list_to_np([1.0, None, 3.0], np.float32, length=3)
    assert arr[0] == 1.0
    assert np.isnan(arr[1])
    assert arr[2] == 3.0


def test_list_to_np_none_input_returns_full_mask() -> None:
    arr = _list_to_np(None, np.int8, length=4)
    assert all(v == np.iinfo(np.int8).min for v in arr)


@pytest.mark.skipif(
    not Path(DEV_ARCHIVE_PATH).exists(),
    reason=f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`",
)
def test_extract_replay_produces_full_schema(tmp_path: Path) -> None:
    """Every column declared in MDS_PER_FRAME_DTYPES is present after extract,
    with the right dtype and a common frame length. Pins schema/extract drift."""
    import py7zr

    with py7zr.SevenZipFile(DEV_ARCHIVE_PATH, "r") as z:
        members = [m for m in z.getnames() if m.endswith(".slp")]
        assert members, "dev archive has no .slp members"
        first = members[0]
        z.extract(path=tmp_path, targets=[first])
    slp = tmp_path / first

    sample = extract_replay(str(slp))
    assert sample is not None, f"extract_replay returned None for {slp}"

    missing = set(MDS_PER_FRAME_DTYPES) - set(sample)
    extra = set(sample) - set(MDS_PER_FRAME_DTYPES)
    assert not missing, f"missing columns: {sorted(missing)[:8]}"
    assert not extra, f"unexpected columns: {sorted(extra)[:8]}"

    frame_len = sample["frame"].shape[0]
    assert frame_len > 0
    for col, dtype in MDS_PER_FRAME_DTYPES.items():
        assert sample[col].shape == (frame_len,), f"length mismatch on {col}"
        assert sample[col].dtype == np.dtype(dtype), f"dtype mismatch on {col}"
