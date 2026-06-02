"""Byte-level tests for the unclosed-.slp finalizer."""

import struct

from hal.data.slp_finalize import finalize_bytes
from hal.data.slp_finalize import finalize_slp
from hal.data.slp_finalize import is_finalized

_HEADER = b"{U\x03raw[$U#l"


def _unfinalized(*, trailing_partial: bool) -> bytes:
    """A minimal Slippi raw stream with rawLength == 0: an EVENT_PAYLOADS event
    declaring one 4-byte command (0x38), two complete 0x38 events, and optionally
    a half-written third one that finalize must drop."""
    event_payloads = bytes([0x35, 0x04, 0x38, 0x00, 0x04])  # 0x35, size=4, {0x38: 4}
    full = bytes([0x38, 1, 2, 3, 4]) + bytes([0x38, 5, 6, 7, 8])
    partial = bytes([0x38, 9, 9]) if trailing_partial else b""
    return _HEADER + struct.pack(">i", 0) + event_payloads + full + partial


def test_finalize_backfills_length_and_appends_footer():
    data = _unfinalized(trailing_partial=False)
    out = finalize_bytes(data)
    assert out is not None
    # rawLength = EVENT_PAYLOADS (5) + two 0x38 events (10) = 15
    assert struct.unpack(">i", out[11:15])[0] == 15
    assert out.endswith(b"U\x08metadata{}}")
    # identical to the input but with rawLength backfilled and the footer appended
    assert out == data[:11] + struct.pack(">i", 15) + data[15:] + b"U\x08metadata{}}"


def test_finalize_drops_half_written_trailing_event():
    out = finalize_bytes(_unfinalized(trailing_partial=True))
    assert out is not None
    # the 3-byte partial 0x38 is dropped: rawLength still 15, footer right after
    assert struct.unpack(">i", out[11:15])[0] == 15
    assert out[15 : 15 + 15] == _unfinalized(trailing_partial=False)[15:]
    assert out.endswith(b"U\x08metadata{}}")


def test_finalize_is_idempotent_on_finalized():
    once = finalize_bytes(_unfinalized(trailing_partial=True))
    assert once is not None
    assert finalize_bytes(once) is None  # already finalized → no-op


def test_finalize_rejects_non_slp():
    assert finalize_bytes(b"not a slippi file at all") is None


def test_finalize_slp_in_place(tmp_path):
    f = tmp_path / "Game.slp"
    f.write_bytes(_unfinalized(trailing_partial=True))
    assert not is_finalized(f)
    assert finalize_slp(f) is True
    assert is_finalized(f)
    assert finalize_slp(f) is False  # second call is a no-op
